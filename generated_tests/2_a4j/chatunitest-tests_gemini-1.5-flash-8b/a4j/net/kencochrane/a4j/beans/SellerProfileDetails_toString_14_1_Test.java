package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class SellerProfileDetails_toString_14_1_Test {

    private SellerProfileDetails sellerProfileDetails;

    private SellerFeedback sellerFeedbackMock;

    @BeforeEach
    void setUp() {
        sellerFeedbackMock = Mockito.mock(SellerFeedback.class);
        sellerProfileDetails = new SellerProfileDetails();
        sellerProfileDetails.setSellerNickname("testNick");
        sellerProfileDetails.setOverallFeedbackRating("4.5");
        sellerProfileDetails.setNumberOfFeedback("100");
        sellerProfileDetails.setNumberofCanceledAuctions("5");
        sellerProfileDetails.setStoreId("123");
        sellerProfileDetails.setStoreName("TestStore");
        sellerProfileDetails.setSellerFeedBack(sellerFeedbackMock);
    }

    @Test
    void testToString() {
        String expectedOutput = "NickName = testNick\n" + "OverallRating = 4.5\n" + "# of feedbacks = 100\n" + "# of Canceled Auctions = 5\n" + "StoreId = 123\n" + "StoreName = TestStore\n" + "FeedBack =\n" + Mockito.mockingDetails(sellerFeedbackMock).toString() + "\n";
        String actualOutput = sellerProfileDetails.toString();
        // Crucial:  Verify the mocked object isn't called unnecessarily
        // added
        Mockito.verifyNoMoreInteractions(sellerFeedbackMock);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testToString_EmptyValues() {
        sellerProfileDetails.setSellerNickname(null);
        sellerProfileDetails.setOverallFeedbackRating("");
        sellerProfileDetails.setNumberOfFeedback(null);
        sellerProfileDetails.setNumberofCanceledAuctions(null);
        sellerProfileDetails.setStoreId("");
        sellerProfileDetails.setStoreName(null);
        String actualOutput = sellerProfileDetails.toString();
        // Assertions for empty/null values in the output - crucial for branch coverage.
        assertTrue(actualOutput.contains("NickName = null") || actualOutput.contains("NickName = "));
        assertTrue(actualOutput.contains("OverallRating = "));
        assertTrue(actualOutput.contains("# of feedbacks = null") || actualOutput.contains("# of feedbacks = "));
        assertTrue(actualOutput.contains("# of Canceled Auctions = null") || actualOutput.contains("# of Canceled Auctions = "));
        assertTrue(actualOutput.contains("StoreId = "));
        assertTrue(actualOutput.contains("StoreName = null") || actualOutput.contains("StoreName = "));
    }
}
