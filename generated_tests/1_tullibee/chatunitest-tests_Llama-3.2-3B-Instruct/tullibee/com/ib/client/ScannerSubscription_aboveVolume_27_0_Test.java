package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_aboveVolume_27_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @Test
    public void testAboveVolume() {
        // Arrange
        int expectedVolume = 10;
        scannerSubscription.aboveVolume(expectedVolume);
        // Act
        int actualVolume = scannerSubscription.aboveVolume();
        // Assert
        assertEquals(expectedVolume, actualVolume);
    }

    @Test
    public void testAboveVolumeWithNegativeValue() {
        // Arrange
        scannerSubscription.aboveVolume(-10);
        // Act and Assert
        int actualVolume = scannerSubscription.aboveVolume();
        assertEquals(-10, actualVolume);
    }

    @Test
    public void testAboveVolumeWithMaxValue() {
        // Arrange
        scannerSubscription.aboveVolume(Integer.MAX_VALUE);
        // Act and Assert
        int actualVolume = scannerSubscription.aboveVolume();
        assertEquals(Integer.MAX_VALUE, actualVolume);
    }

    @Test
    public void testAboveVolumeWithZeroValue() {
        // Arrange
        scannerSubscription.aboveVolume(0);
        // Act and Assert
        int actualVolume = scannerSubscription.aboveVolume();
        assertEquals(0, actualVolume);
    }

    @Test
    public void testAboveVolumeWithZeroValueFromConstructor() {
        // Arrange
        scannerSubscription = new ScannerSubscription();
        scannerSubscription.numberOfRows(-1);
        scannerSubscription.instrument("");
        scannerSubscription.locationCode("");
        scannerSubscription.scanCode("");
        scannerSubscription.abovePrice(0.0);
        scannerSubscription.belowPrice(0.0);
        scannerSubscription.averageOptionVolumeAbove(0);
        scannerSubscription.marketCapAbove(0.0);
        scannerSubscription.marketCapBelow(0.0);
        scannerSubscription.moodyRatingAbove("");
        scannerSubscription.moodyRatingBelow("");
        scannerSubscription.spRatingAbove("");
        scannerSubscription.spRatingBelow("");
        scannerSubscription.maturityDateAbove("");
        scannerSubscription.maturityDateBelow("");
        scannerSubscription.couponRateAbove(0.0);
        scannerSubscription.couponRateBelow(0.0);
        scannerSubscription.excludeConvertible("");
        scannerSubscription.scannerSettingPairs("");
        scannerSubscription.stockTypeFilter("");
        // Act
        int actualVolume = scannerSubscription.aboveVolume();
        // Assert
        assertEquals(0, actualVolume);
    }
}
