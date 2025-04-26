package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class SellerProfileDetails_toString_14_1_Test {

    @Mock
    private SellerFeedback sellerFeedback;

    @InjectMocks
    private SellerProfileDetails focal;

    @Test
    public void testToString() {
        // Arrange
        String nickname = "TestNickname";
        String overallFeedbackRating = "TestRating";
        String numberOfFeedback = "TestFeedback";
        String numberofCanceledAuctions = "TestCanceledAuctions";
        String storeId = "TestStoreId";
        String storeName = "TestStoreName";
        // Act
        String output = focal.toString();
        // Assert
        assertEquals("NickName = TestNickname\n" + "OverallRating = TestRating\n" + "# of feedbacks = " + numberOfFeedback + "\n" + "# of Canceled Auctions = " + numberofCanceledAuctions + "\n" + "StoreId = " + storeId + "\n" + "StoreName = " + storeName + "\n" + "FeedBack = \n" + sellerFeedback + "\n", output);
    }
}
