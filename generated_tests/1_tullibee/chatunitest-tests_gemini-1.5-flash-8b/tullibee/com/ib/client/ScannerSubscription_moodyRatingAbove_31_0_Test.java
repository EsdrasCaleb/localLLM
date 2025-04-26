package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class ScannerSubscription_moodyRatingAbove_31_0_Test {

    @Test
    void testMoodyRatingAbove() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        String rating = "Aaa";
        subscription.moodyRatingAbove(rating);
        Field m_moodyRatingAbove = ScannerSubscription.class.getDeclaredField("m_moodyRatingAbove");
        m_moodyRatingAbove.setAccessible(true);
        assertEquals(rating, m_moodyRatingAbove.get(subscription));
        String nullRating = null;
        subscription.moodyRatingAbove(nullRating);
        assertEquals(nullRating, m_moodyRatingAbove.get(subscription));
        String emptyRating = "";
        subscription.moodyRatingAbove(emptyRating);
        assertEquals(emptyRating, m_moodyRatingAbove.get(subscription));
    }
}
