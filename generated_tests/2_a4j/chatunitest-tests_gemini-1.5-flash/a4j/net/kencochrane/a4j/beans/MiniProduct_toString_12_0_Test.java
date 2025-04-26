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
    void testToString_allFieldsNotNull() {
        MiniProduct product = new MiniProduct();
        product.setAsin("B07XYZ1234");
        product.setName("Test Product");
        product.setManufacturer("Acme Corp");
        product.setPrice("$29.99");
        product.setImageURL("https://example.com/image.jpg");
        product.setProductUrl("https://example.com/product");
        String expected = "B07XYZ1234 \n Test Product \n Acme Corp \n $29.99 \n https://example.com/image.jpg";
        assertEquals(expected, product.toString());
    }

    @Test
    void testToString_someFieldsNull() {
        MiniProduct product = new MiniProduct();
        product.setAsin("B07XYZ1234");
        product.setName("Test Product");
        product.setPrice("$29.99");
        String expected = "B07XYZ1234 \n Test Product \n null \n $29.99 \n null";
        assertEquals(expected, product.toString());
    }

    @Test
    void testToString_allFieldsNull() {
        MiniProduct product = new MiniProduct();
        String expected = "null \n null \n null \n null \n null";
        assertEquals(expected, product.toString());
    }

    @Test
    void testToString_emptyFields() {
        MiniProduct product = new MiniProduct();
        product.setAsin("");
        product.setName("");
        product.setManufacturer("");
        product.setPrice("");
        product.setImageURL("");
        String expected = " \n  \n  \n  \n ";
        assertEquals(expected, product.toString());
    }
}
