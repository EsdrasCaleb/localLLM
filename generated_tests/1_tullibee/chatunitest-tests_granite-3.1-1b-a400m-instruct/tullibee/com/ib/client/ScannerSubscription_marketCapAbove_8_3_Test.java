package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapAbove_8_3_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMarketCapAbove() {
        // Arrange
        scannerSubscription.numberOfRows(5);
        scannerSubscription.instrument("AAPL");
        scannerSubscription.locationCode("NYSE");
        scannerSubscription.scanCode("AAPL");
        scannerSubscription.abovePrice(150.0);
        scannerSubscription.belowPrice(145.0);
        scannerSubscription.aboveVolume(1000);
        scannerSubscription.averageOptionVolumeAbove(100);
        scannerSubscription.marketCapAbove(1000000000.0);
        // Act
        double marketCapAbove = scannerSubscription.marketCapAbove();
        // Assert
        assertEquals(1000000000.0, marketCapAbove);
    }
}
