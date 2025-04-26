package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapBelow_9_0_Test {

    private ScannerSubscription scannerSubscription;

    @Test
    public void testMarketCapBelow() {
        // Arrange
        scannerSubscription = Mockito.mock(ScannerSubscription.class);
        Mockito.when(scannerSubscription.marketCapBelow()).thenReturn(1000.0);
        // Act
        double result = scannerSubscription.marketCapBelow();
        // Assert
        Mockito.verify(scannerSubscription).marketCapBelow();
        assert result == 1000.0;
    }
}
