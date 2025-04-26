package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class MiniProduct_toString_12_0_Test {

    private MiniProduct miniProduct;

    @BeforeEach
    public void setUp() {
        miniProduct = new MiniProduct();
        miniProduct.setAsin("B001234567");
        miniProduct.setName("Sample Product");
        miniProduct.setManufacturer("Sample Manufacturer");
        miniProduct.setPrice("19.99");
        miniProduct.setImageURL("http://example.com/image.jpg");
    }

    @Test
    public void testToString() {
        String expected = "B001234567 \n Sample Product \n Sample Manufacturer \n 19.99 \n http://example.com/image.jpg";
        String actual = miniProduct.toString();
        assertEquals(expected, actual);
    }
}
