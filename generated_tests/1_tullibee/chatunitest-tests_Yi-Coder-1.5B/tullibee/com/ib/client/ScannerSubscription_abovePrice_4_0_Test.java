package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_abovePrice_4_0_Test {

    @Test
    public void test1() {
        ScannerSubscription s = new ScannerSubscription();
        s.instrument("AAPL");
        s.locationCode("US");
        s.scanCode("00001");
        s.abovePrice(100.0);
        s.aboveVolume(100);
        s.averageOptionVolumeAbove(100);
        s.marketCapAbove(1000000000000000000.0);
        s.moodyRatingAbove("AAA");
        s.spRatingAbove("AAA");
        s.maturityDateAbove("2021-01-01");
        s.couponRateAbove(0.01);
        s.excludeConvertible("false");
        s.scannerSettingPairs("123456");
        s.stockTypeFilter("stock");
        double actual = s.abovePrice();
        assertEquals(100.0, actual);
    }
}
