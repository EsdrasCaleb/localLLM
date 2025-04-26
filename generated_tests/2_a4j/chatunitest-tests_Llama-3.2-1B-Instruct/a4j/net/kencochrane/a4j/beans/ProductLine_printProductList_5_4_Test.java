package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ProductLine_printProductList_5_4_Test {

    @Test
    public void testPrintProductList() {
        ProductLine productLine = new ProductLine();
        productLine.setMode("test");
        productLine.setProductInfo(new ProductInfo());
        String expectedOutput = "Mode = test\nProduct list: ";
        String actualOutput = productLine.printProductList();
        assertEquals(expectedOutput, actualOutput);
    }
}
