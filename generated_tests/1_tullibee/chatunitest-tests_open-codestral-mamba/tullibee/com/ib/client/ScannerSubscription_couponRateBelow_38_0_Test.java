package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateBelow_38_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = Mockito.spy(new ScannerSubscription());
    }

    @Test
    public void testCouponRateBelow() {
        double testValue = 5.0;
        scannerSubscription.couponRateBelow(testValue);
        assertEquals(testValue, scannerSubscription.couponRateBelow());
    }
}
