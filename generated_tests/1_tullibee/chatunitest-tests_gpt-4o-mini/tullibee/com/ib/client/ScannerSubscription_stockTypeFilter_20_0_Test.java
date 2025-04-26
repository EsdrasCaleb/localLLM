package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_stockTypeFilter_20_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testStockTypeFilter_ReturnsDefault_WhenNotSet() {
        // Act
        String result = scannerSubscription.stockTypeFilter();
        // Assert
        assertEquals(null, result);
    }

    @Test
    public void testStockTypeFilter_ReturnsValue_WhenSet() {
        // Arrange
        String expectedStockType = "Equity";
        scannerSubscription.stockTypeFilter(expectedStockType);
        // Act
        String result = scannerSubscription.stockTypeFilter();
        // Assert
        assertEquals(expectedStockType, result);
    }
}
