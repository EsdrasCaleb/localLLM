package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateAbove_16_1_Test {

    @Test
    public void testCouponRateAbove_ReturnsInitialValue() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        double result = scannerSubscription.couponRateAbove();
        assertEquals(Double.MAX_VALUE, result);
    }

    @Test
    public void testCouponRateAbove_ChangesValue() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.couponRateAbove(10.0);
        double result = scannerSubscription.couponRateAbove();
        assertEquals(10.0, result);
    }
}
