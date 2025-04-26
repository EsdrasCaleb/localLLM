package com.ib.client;

import java.lang.reflect.Field;
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
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testCouponRateBelow() throws NoSuchFieldException, IllegalAccessException {
        double testCouponRateBelow = 3.5;
        // Invoke the method under test
        scannerSubscription.couponRateBelow(testCouponRateBelow);
        // Use reflection to access the private field and verify its value
        Field couponRateBelowField = ScannerSubscription.class.getDeclaredField("m_couponRateBelow");
        couponRateBelowField.setAccessible(true);
        double actualCouponRateBelow = (double) couponRateBelowField.get(scannerSubscription);
        assertEquals(testCouponRateBelow, actualCouponRateBelow, "The couponRateBelow should be set correctly");
    }
}
