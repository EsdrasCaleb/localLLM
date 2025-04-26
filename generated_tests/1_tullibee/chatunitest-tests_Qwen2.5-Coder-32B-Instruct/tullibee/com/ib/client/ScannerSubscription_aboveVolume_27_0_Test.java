package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_aboveVolume_27_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testAboveVolumeSetsCorrectValue() throws NoSuchFieldException, IllegalAccessException {
        // Given
        int testVolume = 1000;
        Field aboveVolumeField = ScannerSubscription.class.getDeclaredField("m_aboveVolume");
        aboveVolumeField.setAccessible(true);
        // When
        scannerSubscription.aboveVolume(testVolume);
        // Then
        assertEquals(testVolume, aboveVolumeField.get(scannerSubscription));
    }

    @Test
    public void testAboveVolumeSetsMaxValue() throws NoSuchFieldException, IllegalAccessException {
        // Given
        int testVolume = Integer.MAX_VALUE;
        Field aboveVolumeField = ScannerSubscription.class.getDeclaredField("m_aboveVolume");
        aboveVolumeField.setAccessible(true);
        // When
        scannerSubscription.aboveVolume(testVolume);
        // Then
        assertEquals(testVolume, aboveVolumeField.get(scannerSubscription));
    }

    @Test
    public void testAboveVolumeSetsZeroValue() throws NoSuchFieldException, IllegalAccessException {
        // Given
        int testVolume = 0;
        Field aboveVolumeField = ScannerSubscription.class.getDeclaredField("m_aboveVolume");
        aboveVolumeField.setAccessible(true);
        // When
        scannerSubscription.aboveVolume(testVolume);
        // Then
        assertEquals(testVolume, aboveVolumeField.get(scannerSubscription));
    }

    @Test
    public void testAboveVolumeSetsNegativeValue() throws NoSuchFieldException, IllegalAccessException {
        // Given
        int testVolume = -1;
        Field aboveVolumeField = ScannerSubscription.class.getDeclaredField("m_aboveVolume");
        aboveVolumeField.setAccessible(true);
        // When
        scannerSubscription.aboveVolume(testVolume);
        // Then
        assertEquals(testVolume, aboveVolumeField.get(scannerSubscription));
    }
}
