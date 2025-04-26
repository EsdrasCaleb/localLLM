package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_averageOptionVolumeAbove_28_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testAverageOptionVolumeAbove() throws NoSuchFieldException, IllegalAccessException {
        int testVolume = 1000;
        scannerSubscription.averageOptionVolumeAbove(testVolume);
        Field field = ScannerSubscription.class.getDeclaredField("m_averageOptionVolumeAbove");
        field.setAccessible(true);
        int result = (int) field.get(scannerSubscription);
        assertEquals(testVolume, result, "The average option volume above should be set to the provided volume.");
    }
}
