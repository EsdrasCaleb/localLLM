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

class FullProduct_printFullProduct_8_0_Test {

    @Mock
    private ProductDetails details;

    @InjectMocks
    private FullProduct fullProduct;

    private final ByteArrayOutputStream outputStream = new ByteArrayOutputStream();

    private final PrintStream originalOut = System.out;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.initMocks(this);
        System.setOut(new PrintStream(outputStream));
    }

    @Test
    void testPrintFullProduct() {
        when(details.toString()).thenReturn("Product Details");
        fullProduct.setAccessories(new ArrayList<>());
        fullProduct.setSimilarItems(new ArrayList<>());
        fullProduct.printFullProduct();
        String expectedOutput = "Product Details\n\n-- Accessories --\n\n-- Similar Products --\n";
        assertEquals(expectedOutput, outputStream.toString());
    }

    @AfterEach
    void tearDown() {
        System.setOut(originalOut);
    }
}
