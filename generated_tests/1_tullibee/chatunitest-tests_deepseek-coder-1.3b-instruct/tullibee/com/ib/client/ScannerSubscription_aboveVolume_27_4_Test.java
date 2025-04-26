package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_aboveVolume_27_4_Test {

    private ScannerSubscription scannerSubscription;

    @Mock
    private ScannerSubscription mockScannerSubscription;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        scannerSubscription = new ScannerSubscription();
        scannerSubscription.numberOfRows(10);
        scannerSubscription.instrument("TEST");
        scannerSubscription.locationCode("LOCATION");
        scannerSubscription.scanCode("SCANCODE");
        scannerSubscription.abovePrice(100.0);
        scannerSubscription.belowPrice(90.0);
        scannerSubscription.aboveVolume(20);
        scannerSubscription.averageOptionVolumeAbove(15);
        scannerSubscription.marketCapAbove(1000.0);
        scannerSubscription.marketCapBelow(900.0);
        scannerSubscription.moodyRatingAbove("A+");
        scannerSubscription.moodyRatingBelow("A-");
        scannerSubscription.spRatingAbove("A+");
        scannerSubscription.spRatingBelow("A-");
        scannerSubscription.maturityDateAbove("2020-01-01");
        scannerSubscription.maturityDateBelow("2020-02-01");
        scannerSubscription.couponRateAbove(0.05);
        scannerSubscription.couponRateBelow(0.04);
        scannerSubscription.excludeConvertible("N");
        scannerSubscription.scannerSettingPairs("SETTING");
        scannerSubscription.stockTypeFilter("TYPE");
    }

    @Test
    public void testAboveVolume() {
        int expectedVolume = 20;
        scannerSubscription.aboveVolume(expectedVolume);
        int actualVolume = scannerSubscription.aboveVolume();
        assertEquals(expectedVolume, actualVolume);
    }
}
