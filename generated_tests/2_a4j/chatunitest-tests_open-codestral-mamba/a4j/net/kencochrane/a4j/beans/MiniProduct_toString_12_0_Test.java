package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class MiniProduct_toString_12_0_Test {

    @InjectMocks
    private MiniProduct miniProduct;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        miniProduct = new MiniProduct();
    }

    @Test
    public void testToString() {
        miniProduct.setAsin("ASIN123");
        miniProduct.setName("Product Name");
        miniProduct.setManufacturer("Manufacturer Name");
        miniProduct.setPrice("$100");
        miniProduct.setImageURL("http://imageurl.com");
        String expectedToString = "ASIN123 \n Product Name \n Manufacturer Name \n $100 \n http://imageurl.com";
        assertEquals(expectedToString, miniProduct.toString());
    }
}
