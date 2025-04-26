package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class ProductInfo_printProductList_10_3_Test {

    @Test
    void testPrintProductList() {
        // Arrange
        ProductInfo productInfo = new ProductInfo();
        // Act
        String expectedOutput = "Total results = 100 \nTotal pages = 200 \n";
        // Assert
        assertEquals(expectedOutput, productInfo.printProductList());
    }
}
