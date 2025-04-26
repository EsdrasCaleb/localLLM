package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_stockTypeFilter_20_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testStockTypeFilter() {
        // Arrange
        String expectedStockTypeFilter = "TestFilter";
        scannerSubscription.stockTypeFilter(expectedStockTypeFilter);
        // Act
        String actualStockTypeFilter = scannerSubscription.stockTypeFilter();
        // Assert
        assertEquals(expectedStockTypeFilter, actualStockTypeFilter);
    }
}
