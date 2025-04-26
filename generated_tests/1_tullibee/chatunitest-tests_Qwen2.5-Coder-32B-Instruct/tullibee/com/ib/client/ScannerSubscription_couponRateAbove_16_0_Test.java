package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateAbove_16_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testCouponRateAbove_DefaultValue() {
        // Default value of m_couponRateAbove is Double.MAX_VALUE
        assertEquals(Double.MAX_VALUE, scannerSubscription.couponRateAbove());
    }

    @Test
    public void testCouponRateAbove_SetValue() throws NoSuchFieldException, IllegalAccessException {
        // Set a new value using reflection
        Field couponRateAboveField = ScannerSubscription.class.getDeclaredField("m_couponRateAbove");
        couponRateAboveField.setAccessible(true);
        double testValue = 5.75;
        couponRateAboveField.set(scannerSubscription, testValue);
        // Verify the getter returns the set value
        assertEquals(testValue, scannerSubscription.couponRateAbove());
    }
}
