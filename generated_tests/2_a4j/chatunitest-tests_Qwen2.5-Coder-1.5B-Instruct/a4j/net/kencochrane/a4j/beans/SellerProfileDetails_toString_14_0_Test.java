package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class SellerProfileDetails_toString_14_0_Test {

    @Test
    public void testToString() throws Exception {
        // Create a mock object for SellerFeedBack
        SellerFeedback mockFeedback = mock(SellerFeedback.class);
        when(mockFeedback.toString()).thenReturn("Mocked Feedback");
        // Create an instance of SellerProfileDetails and set its properties
        SellerProfileDetails sellerProfile = new SellerProfileDetails();
        sellerProfile.setSellerNickname("JohnDoe");
        sellerProfile.setOverallFeedbackRating("4.5");
        sellerProfile.setNumberOfFeedback("10");
        sellerProfile.setNumberofCanceledAuctions("2");
        sellerProfile.setStoreId("12345");
        sellerProfile.setStoreName("Example Store");
        sellerProfile.setSellerFeedBack(mockFeedback);
        // Call the toString() method on the sellerProfile object
        String result = sellerProfile.toString();
        // Assert the expected result
        assertEquals("NickName = JohnDoe\n" + "OverallRating = 4.5\n" + "# of feedbacks = 10\n" + "# of Canceled Auctions = 2\n" + "StoreId = 12345\n" + "StoreName = Example Store\n" + "FeedBack = \nMocked Feedback\n", result);
    }
}
