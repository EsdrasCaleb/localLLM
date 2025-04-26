package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SellerProfileDetails_toString_14_0_Test {

    private SellerProfileDetails sellerProfileDetails;

    private SellerFeedback mockSellerFeedback;

    @BeforeEach
    public void setUp() {
        sellerProfileDetails = new SellerProfileDetails();
        mockSellerFeedback = Mockito.mock(SellerFeedback.class);
        sellerProfileDetails.setSellerNickname("TestNickname");
        sellerProfileDetails.setOverallFeedbackRating("4.5");
        sellerProfileDetails.setNumberOfFeedback("100");
        sellerProfileDetails.setNumberofCanceledAuctions("5");
        sellerProfileDetails.setStoreId("12345");
        sellerProfileDetails.setStoreName("TestStore");
        sellerProfileDetails.setSellerFeedBack(mockSellerFeedback);
        // Mocking the toString method of SellerFeedback
        Mockito.when(mockSellerFeedback.toString()).thenReturn("Mocked Feedback Details");
    }

    @Test
    public void testToString() {
        String expectedOutput = "NickName = TestNickname\n" + "OverallRating = 4.5\n" + "# of feedbacks = 100\n" + "# of Canceled Auctions = 5\n" + "StoreId = 12345\n" + "StoreName = TestStore\n" + "FeedBack = \nMocked Feedback Details\n";
        String actualOutput = sellerProfileDetails.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
