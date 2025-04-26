package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class MiniProduct_toString_12_0_Test {

    @Test
    public void testToString() {
        MiniProduct product = new MiniProduct();
        product.setName("Test Product");
        product.setManufacturer("Test Manufacturer");
        product.setPrice("10.99");
        product.setAsin("1234567890");
        product.setImageURL("https://example.com/image.jpg");
        String expectedString = "1234567890\nTest Product\nTest Manufacturer\n10.99\nhttps://example.com/image.jpg";
        String actualString = product.toString();
        assertEquals(expectedString, actualString);
    }

    @Test
    public void testToString_NullName() {
        MiniProduct product = new MiniProduct();
        product.setAsin("1234567890");
        product.setPrice("10.99");
        product.setImageURL("https://example.com/image.jpg");
        String expectedString = "1234567890\n\n\n\nhttps://example.com/image.jpg";
        String actualString = product.toString();
        assertEquals(expectedString, actualString);
    }

    @Test
    public void testToString_NullManufacturer() {
        MiniProduct product = new MiniProduct();
        product.setName("Test Product");
        product.setPrice("10.99");
        product.setAsin("1234567890");
        product.setImageURL("https://example.com/image.jpg");
        String expectedString = "1234567890\nTest Product\n\n10.99\nhttps://example.com/image.jpg";
        String actualString = product.toString();
        assertEquals(expectedString, actualString);
    }

    @Test
    public void testToString_NullPrice() {
        MiniProduct product = new MiniProduct();
        product.setName("Test Product");
        product.setManufacturer("Test Manufacturer");
        product.setAsin("1234567890");
        product.setImageURL("https://example.com/image.jpg");
        String expectedString = "1234567890\nTest Product\nTest Manufacturer\n\nhttps://example.com/image.jpg";
        String actualString = product.toString();
        assertEquals(expectedString, actualString);
    }

    @Test
    public void testToString_NullImageURL() {
        MiniProduct product = new MiniProduct();
        product.setName("Test Product");
        product.setManufacturer("Test Manufacturer");
        product.setAsin("1234567890");
        product.setPrice("10.99");
        String expectedString = "1234567890\nTest Product\nTest Manufacturer\n10.99\n";
        String actualString = product.toString();
        assertEquals(expectedString, actualString);
    }

    @Test
    public void testToString_EmptyFields() {
        MiniProduct product = new MiniProduct();
        String expectedString = "";
        String actualString = product.toString();
        assertEquals(expectedString, actualString);
    }
}
