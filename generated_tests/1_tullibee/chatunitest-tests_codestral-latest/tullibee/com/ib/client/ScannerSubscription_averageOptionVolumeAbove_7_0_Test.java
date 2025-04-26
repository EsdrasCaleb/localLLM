package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_averageOptionVolumeAbove_7_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testAverageOptionVolumeAbove() throws Exception {
        // Set the private field m_averageOptionVolumeAbove using reflection
        Field field = ScannerSubscription.class.getDeclaredField("m_averageOptionVolumeAbove");
        field.setAccessible(true);
        field.set(scannerSubscription, 100);
        // Call the method under test
        int result = scannerSubscription.averageOptionVolumeAbove();
        // Assert the result
        assertEquals(100, result);
    }

    @Test
    public void testAverageOptionVolumeAbove_DefaultValue() {
        // Call the method under test
        int result = scannerSubscription.averageOptionVolumeAbove();
        // Assert the result
        assertEquals(Integer.MAX_VALUE, result);
    }
}
