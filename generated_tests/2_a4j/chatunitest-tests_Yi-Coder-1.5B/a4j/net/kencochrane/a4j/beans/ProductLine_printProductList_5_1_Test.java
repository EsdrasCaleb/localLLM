package net.kencochrane.a4j.beans;

import static org.junit.Assert.assertEquals;
import org.junit.Test;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ProductLine_printProductList_5_1_Test {

    @Test
    public void testPrintProductList() {
        ProductLine productLine = new ProductLine();
        productLine.setMode("PROD");
        productLine.setProductInfo(new ProductInfo());
        String actual = productLine.printProductList();
        String expected = "Mode = PROD\n" + "\n";
        assertEquals(expected, actual);
    }
}
