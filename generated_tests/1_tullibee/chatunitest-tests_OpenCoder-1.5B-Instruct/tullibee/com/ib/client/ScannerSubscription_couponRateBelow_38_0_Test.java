package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateBelow_38_0_Test {

    @Test
    public void testCouponRateBelow() {
        ScannerSubscription scannerSubscription = Mockito.mock(ScannerSubscription.class);
        double newCouponRateBelow = 1.5;
        scannerSubscription.couponRateBelow(newCouponRateBelow);
        Mockito.verify(scannerSubscription).couponRateBelow(newCouponRateBelow);
    }
}
