package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class ProductLine_toString_4_0_Test {

    private ProductLine productLine;

    private ProductInfo productInfo;

    @BeforeEach
    void setUp() {
        productInfo = Mockito.mock(ProductInfo.class);
        productLine = new ProductLine();
        try {
            Field productInfoField = ProductLine.class.getDeclaredField("productInfo");
            productInfoField.setAccessible(true);
            productInfoField.set(productLine, productInfo);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            // Include exception for better debugging
            fail("Error accessing private field.", e);
        }
    }

    @Test
    void testToString_productInfoNotNull() {
        when(productInfo.toString()).thenReturn("Product Info: Test");
        String expected = "ProductLine{productInfo='Product Info: Test'}";
        String actual = productLine.toString();
        assertEquals(expected, actual);
    }

    @Test
    void testToString_productInfoNull() {
        try {
            Field productInfoField = ProductLine.class.getDeclaredField("productInfo");
            productInfoField.setAccessible(true);
            productInfoField.set(productLine, null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private field.", e);
        }
        String expected = "ProductLine{productInfo=null}";
        String actual = productLine.toString();
        assertEquals(expected, actual);
    }

    static class ProductLine {

        private ProductInfo productInfo;

        @Override
        public String toString() {
            return "ProductLine{productInfo='" + ((productInfo != null) ? productInfo.toString() : "null") + "'}";
        }
    }

    static class ProductInfo {

        @Override
        public String toString() {
            return "Product Info: Test";
        }
    }

    @Test
    void testToString_validModeAndProductInfo() {
        String mode = "testMode";
        try {
            Field modeField = ProductLine.class.getDeclaredField("mode");
            modeField.setAccessible(true);
            modeField.set(productLine, mode);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private field.");
        }
        Mockito.when(productInfo.toString()).thenReturn("Product Info: Test");
        String expectedOutput = "Mode = testMode\nProduct Info: Test\n";
        String actualOutput = productLine.toString();
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testToString_nullMode() {
        try {
            Field modeField = ProductLine.class.getDeclaredField("mode");
            modeField.setAccessible(true);
            modeField.set(productLine, null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private field.");
        }
        Mockito.when(productInfo.toString()).thenReturn("Product Info: Test");
        String expectedOutput = "Mode = null\nProduct Info: Test\n";
        String actualOutput = productLine.toString();
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testToString_nullProductInfo() {
        try {
            Field productInfoField = ProductLine.class.getDeclaredField("productInfo");
            productInfoField.setAccessible(true);
            productInfoField.set(productLine, null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private field.");
        }
        String expectedOutput = "Mode = null\nnull\n";
        String actualOutput = productLine.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
