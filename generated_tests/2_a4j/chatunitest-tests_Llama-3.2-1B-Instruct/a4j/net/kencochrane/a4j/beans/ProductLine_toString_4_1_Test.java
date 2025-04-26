package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ProductLine_toString_4_1_Test {

    @Test
    public void testToString() {
        ProductLine productLine = new ProductLine();
        productLine.setMode("mode1");
        productLine.setProductInfo(new ProductInfo());
        String expectedOutput = "Mode = mode1\nProductInfo: ProductInfo{ mode='mode1', product='product1' }\n";
        String actualOutput = productLine.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
