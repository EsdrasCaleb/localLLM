package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SellerProfileDetails_toString_14_0_Test {

    @Test
    public void testToString() {
        SellerProfileDetails seller = new SellerProfileDetails();
        seller.setSellerNickname("testNickname");
        seller.setOverallFeedbackRating("5.0");
        seller.setNumberOfFeedback("10");
        seller.setNumberofCanceledAuctions("2");
        seller.setStoreId("12345");
        seller.setStoreName("testStore");
        SellerFeedback sellerFeedback = mock(SellerFeedback.class);
        when(sellerFeedback.toString()).thenReturn("Feedback details");
        seller.setSellerFeedBack(sellerFeedback);
        String expected = "NickName = testNickname\nOverallRating = 5.0\n# of feedbacks = 10\n# of Canceled Auctions = 2\nStoreId = 12345\nStoreName = testStore\nFeedBack = \nFeedback details\n";
        assertEquals(expected, seller.toString());
    }
}
