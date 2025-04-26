package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateAbove_16_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testCouponRateAbove() {
        // Set the private field m_couponRateAbove using reflection
        try {
            java.lang.reflect.Field field = ScannerSubscription.class.getDeclaredField("m_couponRateAbove");
            field.setAccessible(true);
            field.set(scannerSubscription, 5.5);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
        // Call the method under test
        double result = scannerSubscription.couponRateAbove();
        // Verify the result
        assertEquals(5.5, result, 0.001);
    }
}
