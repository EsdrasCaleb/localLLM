package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_aboveVolume_6_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testAboveVolume_DefaultValue() throws NoSuchFieldException, IllegalAccessException {
        // Given the default value of m_aboveVolume is Integer.MAX_VALUE
        Field aboveVolumeField = ScannerSubscription.class.getDeclaredField("m_aboveVolume");
        aboveVolumeField.setAccessible(true);
        aboveVolumeField.set(scannerSubscription, Integer.MAX_VALUE);
        // When aboveVolume() is called
        int result = scannerSubscription.aboveVolume();
        // Then it should return Integer.MAX_VALUE
        assertEquals(Integer.MAX_VALUE, result);
    }

    @Test
    public void testAboveVolume_SetValue() throws NoSuchFieldException, IllegalAccessException {
        // Given a specific value is set for m_aboveVolume
        int testValue = 1000;
        Field aboveVolumeField = ScannerSubscription.class.getDeclaredField("m_aboveVolume");
        aboveVolumeField.setAccessible(true);
        aboveVolumeField.set(scannerSubscription, testValue);
        // When aboveVolume() is called
        int result = scannerSubscription.aboveVolume();
        // Then it should return the set value
        assertEquals(testValue, result);
    }

    @Test
    public void testAboveVolume_SetAnotherValue() throws NoSuchFieldException, IllegalAccessException {
        // Given another specific value is set for m_aboveVolume
        int testValue = 5000;
        Field aboveVolumeField = ScannerSubscription.class.getDeclaredField("m_aboveVolume");
        aboveVolumeField.setAccessible(true);
        aboveVolumeField.set(scannerSubscription, testValue);
        // When aboveVolume() is called
        int result = scannerSubscription.aboveVolume();
        // Then it should return the new set value
        assertEquals(testValue, result);
    }
}
