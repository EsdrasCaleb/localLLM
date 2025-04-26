package net.kencochrane.a4j.beans;

import java.util.Objects;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class MiniProduct_toString_12_0_Test {

    private MiniProduct miniProduct;

    @Test
    public void testToString() {
        miniProduct = new MiniProduct();
        miniProduct.setAsin("1234567890");
        miniProduct.setName("Test Product");
        miniProduct.setManufacturer("Test Manufacturer");
        miniProduct.setPrice("19.99");
        miniProduct.setImageURL("http://example.com/image.jpg");
        miniProduct.setProductUrl("http://example.com/product.html");
        String expected = "1234567890 \n Test Product \n Test Manufacturer \n 19.99 \n http://example.com/image.jpg";
        String actual = miniProduct.toString();
        assertEquals(expected, actual);
    }
}
