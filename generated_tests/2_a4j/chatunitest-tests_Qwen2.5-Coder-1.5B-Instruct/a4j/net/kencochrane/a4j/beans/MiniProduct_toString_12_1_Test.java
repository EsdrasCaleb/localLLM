package net.kencochrane.a4j.beans;

import net.kencochrane.a4j.beans.MiniProduct;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class MiniProduct_toString_12_1_Test {

    @Test
    public void testToString() throws Exception {
        // Create a mock instance of MiniProduct
        MiniProduct miniProduct = mock(MiniProduct.class);
        // Set up expected values
        when(miniProduct.getName()).thenReturn("Example Product");
        when(miniProduct.getManufacturer()).thenReturn("Example Manufacturer");
        when(miniProduct.getPrice()).thenReturn("10.99");
        when(miniProduct.getAsin()).thenReturn("B076W2G4T3");
        when(miniProduct.getImageURL()).thenReturn("https://example.com/image.jpg");
        when(miniProduct.getProductUrl()).thenReturn("https://example.com/product");
        // Call the method under test
        String result = miniProduct.toString();
        // Correctly call assertEquals from junit.jupiter.api.Assertions
        assertEquals("B076W2G4T3 \n Example Product \n Example Manufacturer \n 10.99 \n https://example.com/image.jpg", result);
    }
}
