package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateAbove_16_0_Test {

    @Test
    void testCouponRateAbove() {
        ScannerSubscription scannerSubscription = mock(ScannerSubscription.class);
        when(scannerSubscription.couponRateAbove()).thenReturn(10.0);
        double result = scannerSubscription.couponRateAbove();
        assertEquals(10.0, result, 0.0);
    }
}
