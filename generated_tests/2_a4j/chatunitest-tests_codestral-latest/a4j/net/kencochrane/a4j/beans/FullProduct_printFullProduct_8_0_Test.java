package net.kencochrane.a4j.beans;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class FullProduct_printFullProduct_8_0_Test {

    @InjectMocks
    private FullProduct fullProduct;

    @Mock
    private ProductDetails details;

    @Mock
    private ArrayList<String> accessories;

    @Mock
    private ArrayList<String> similarItems;

    private final ByteArrayOutputStream outContent = new ByteArrayOutputStream();

    private final PrintStream originalOut = System.out;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        System.setOut(new PrintStream(outContent));
    }

    @Test
    public void testPrintFullProduct() {
        // Arrange
        when(details.toString()).thenReturn("Product Details");
        when(accessories.size()).thenReturn(2);
        when(accessories.get(0)).thenReturn("Accessory 1");
        when(accessories.get(1)).thenReturn("Accessory 2");
        when(similarItems.size()).thenReturn(2);
        when(similarItems.get(0)).thenReturn("Similar Item 1");
        when(similarItems.get(1)).thenReturn("Similar Item 2");
        fullProduct.setDetails(details);
        fullProduct.setAccessories(accessories);
        fullProduct.setSimilarItems(similarItems);
        // Act
        fullProduct.printFullProduct();
        // Assert
        String expectedOutput = "Product Details\n\n-- Accessories --\nAccessory 1\n\nAccessory 2\n\n-- Similar Products --\nSimilar Item 1\n\nSimilar Item 2\n\n";
        assertEquals(expectedOutput, outContent.toString());
        // Clean up
        System.setOut(originalOut);
    }
}
