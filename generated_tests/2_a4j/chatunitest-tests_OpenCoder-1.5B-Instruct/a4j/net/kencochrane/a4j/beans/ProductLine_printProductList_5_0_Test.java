package net.kencochrane.a4j.beans;

import java.util.Optional;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ProductLine_printProductList_5_0_Test {

    @Test
    public void testPrintProductList() {
        // Create a mock object of ProductInfo
        ProductInfo productInfoMock = Mockito.mock(ProductInfo.class);
        // Create an instance of ProductLine with a mock object of ProductInfo
        ProductLine productLine = new ProductLine();
        productLine.productInfo = productInfoMock;
        // Set the mode
        productLine.mode = "Test Mode";
        // Mock the printProductList() method of ProductInfo
        Mockito.when(productInfoMock.printProductList()).thenReturn("Mocked Product List\n");
        // Call the method being tested
        String result = productLine.printProductList();
        // Verify that the printProductList() method of ProductInfo was called
        Mockito.verify(productInfoMock).printProductList();
        // Verify that the mode was set correctly
        assert "Mode = Test Mode".equals(result);
    }
}
