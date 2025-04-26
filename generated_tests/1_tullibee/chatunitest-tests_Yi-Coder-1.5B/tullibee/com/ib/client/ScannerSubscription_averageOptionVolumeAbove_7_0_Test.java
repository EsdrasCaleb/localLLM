package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_averageOptionVolumeAbove_7_0_Test {

    // Test class
    @Test
    public void testAverageOptionVolumeAbove() {
        ScannerSubscription scanner = new ScannerSubscription();
        scanner.numberOfRows(1);
        scanner.instrument("AAPL");
        scanner.locationCode("USA");
        scanner.scanCode("1");
        scanner.abovePrice(100);
        scanner.averageOptionVolumeAbove(100);
        scanner.aboveVolume(100);
        scanner.marketCapAbove(100);
        scanner.moodyRatingAbove("A");
        scanner.spRatingAbove("A");
        scanner.maturityDateAbove("1/1/2023");
        scanner.couponRateAbove(0.05);
        scanner.excludeConvertible("Y");
        scanner.scannerSettingPairs("Y");
        scanner.stockTypeFilter("AAPL");
        int actual = scanner.averageOptionVolumeAbove();
        assertEquals(100, actual);
    }
}
