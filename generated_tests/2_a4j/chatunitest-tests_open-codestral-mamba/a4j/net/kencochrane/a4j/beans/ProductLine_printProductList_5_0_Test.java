package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class ProductLine_printProductList_5_0_Test {

    @Mock
    private ProductInfo productInfo;

    @InjectMocks
    private ProductLine productLine;

    @BeforeEach
    public void setUp() {
        productLine.setMode("Test Mode");
    }

    @Test
    public void testPrintProductList() {
        String expectedOutput = "Mode = Test Mode\n" + "Product List\n";
        when(productInfo.printProductList()).thenReturn("Product List");
        String actualOutput = productLine.printProductList();
        assertEquals(expectedOutput, actualOutput);
    }
}
