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
        MiniProduct miniProduct = new MiniProduct();
        miniProduct.setAsin("12345");
        miniProduct.setName("Product Name");
        miniProduct.setManufacturer("Product Manufacturer");
        miniProduct.setPrice("9.99");
        miniProduct.setImageURL("https://example.com/image.jpg");
        miniProduct.setProductUrl("https://example.com/product");
        String expected = "12345 \n Product Name \n Product Manufacturer \n 9.99 \n https://example.com/image.jpg";
        String actual = miniProduct.toString();
        assertEquals(expected, actual);
    }
}
