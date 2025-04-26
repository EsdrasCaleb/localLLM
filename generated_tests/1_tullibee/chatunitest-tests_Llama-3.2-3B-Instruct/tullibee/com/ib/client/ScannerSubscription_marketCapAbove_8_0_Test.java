package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapAbove_8_0_Test {

    @Test
    public void testMarketCapAbove_ReturnsDoubleMaxValue_WhenMarketCapAboveNotSet() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        double result = scannerSubscription.marketCapAbove();
        assertEquals(Double.MAX_VALUE, result);
    }

    @Test
    public void testMarketCapAbove_ReturnsMarketCapAboveValue_WhenMarketCapAboveSet() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.marketCapAbove(100.0);
        double result = scannerSubscription.marketCapAbove();
        assertEquals(100.0, result, 0.01);
    }

    @Test
    public void testMarketCapAbove_ThrowsNullPointerException_WhenScannerSubscriptionIsNull() {
        assertThrows(NullPointerException.class, () -> new ScannerSubscription().marketCapAbove());
    }
}
