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
        String expected = "Mode = \n";
        String actual = productLine.toString();
        assertEquals(expected, actual);
    }

    @Test
    public void testToStringWithMode() {
        ProductLine productLine = new ProductLine();
        productLine.setMode("Test Mode");
        String expected = "Mode = Test Mode\n";
        String actual = productLine.toString();
        assertEquals(expected, actual);
    }

    @Test
    public void testToStringWithProductInfo() {
        ProductInfo productInfo = new ProductInfo();
        ProductLine productLine = new ProductLine();
        productLine.setProductInfo(productInfo);
        String expected = "Mode = \n" + productInfo.toString();
        String actual = productLine.toString();
        assertEquals(expected, actual);
    }

    @Test
    public void testToStringWithModeAndProductInfo() {
        ProductInfo productInfo = new ProductInfo();
        ProductLine productLine = new ProductLine();
        productLine.setMode("Test Mode");
        productLine.setProductInfo(productInfo);
        String expected = "Mode = Test Mode\n" + productInfo.toString();
        String actual = productLine.toString();
        assertEquals(expected, actual);
    }

    @Test
    public void testToStringWithEmptyMode() {
        ProductLine productLine = new ProductLine();
        productLine.setMode("");
        String expected = "Mode = \n";
        String actual = productLine.toString();
        assertEquals(expected, actual);
    }

    @Test
    public void testToStringWithEmptyProductInfo() {
        ProductInfo productInfo = new ProductInfo();
        ProductLine productLine = new ProductLine();
        productLine.setProductInfo(productInfo);
        String expected = "Mode = \n" + productInfo.toString();
        String actual = productLine.toString();
        assertEquals(expected, actual);
    }
}
