package utils
import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types._

import scala.collection.mutable.ListBuffer

/**
 * 自定义聚合函数
 *

 */
class Middle extends UserDefinedAggregateFunction {

  /**
   * 分割字符串
   */
  val split_str = "_"

  // 输入值 类型
  override def inputSchema: StructType = StructType(StructField("data", DoubleType) :: Nil)

  // 缓冲类型
  override def bufferSchema: StructType = StructType(StructField("middle", StringType) :: Nil)

  // 返回值类型
  override def dataType: DataType = DoubleType

  //对于数据一样的情况下 返回值时候一样
  override def deterministic: Boolean = true

  /**
   * 初始化时调用
   *
   * @param buffer
   */
  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer.update(0, "")
  }

  /**
   * 一个节点统计操作，每次输入一行记录。需要根据旧的缓冲和新来的数据 做逻辑处理
   *
   * @param buffer 缓冲引用
   * @param input  新的值
   */
  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    buffer.update(0, buffer.get(0).asInstanceOf[String] + split_str + input.getDouble(0).toString)
  }

  /**
   * 多条记录时如何处理 -》 其实就是两个Node计算出来的结果合并操作
   *
   * @param buffer1 节点一的缓冲区
   * @param buffer2 节点二缓冲区
   */
  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    buffer1.update(0, buffer1.get(0).asInstanceOf[String] + split_str + buffer2.get(0).asInstanceOf[String])
  }

  /**
   * 最后输出 即 函数输出。 这里作用主要是取中位数。
   *
   * @param buffer 汇集后的缓冲区
   * @return
   */
  override def evaluate(buffer: Row): Any = {

    val str = buffer.get(0).asInstanceOf[String]
    val arrays = str.split(split_str)
    val list = new ListBuffer[Double]
    for (str <- arrays) {
      if (str != null && !str.isEmpty) {
        list.append(str.toDouble)
      }
    }
    if (list.isEmpty) {
      return null
    }
    val sorted = list.sorted
    var size = sorted.size
    size = sorted.size
    // 偶数
    if (size % 2 == 0) {
      val middle_first = size / 2
      val middle_second = (size / 2) - 1
      (sorted(middle_first) + sorted(middle_second)) / 2
    } else {
      sorted(size / 2)
    }
  }
}
