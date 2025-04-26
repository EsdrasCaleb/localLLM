package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateBelow_17_0_Test {

    @Test
    void couponRateBelow_shouldReturnCorrectValue() {
        ScannerSubscription scannerSubscription = mock(ScannerSubscription.class);
        when(scannerSubscription.couponRateBelow()).thenReturn(10.5);
        double result = scannerSubscription.couponRateBelow();
        assert result == 10.5;
    }
}
