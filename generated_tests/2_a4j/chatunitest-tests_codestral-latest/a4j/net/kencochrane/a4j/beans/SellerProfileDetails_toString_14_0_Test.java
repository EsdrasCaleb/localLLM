package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class SellerProfileDetails_toString_14_0_Test {

    @InjectMocks
    private SellerProfileDetails sellerProfileDetails;

    @Mock
    private SellerFeedback sellerFeedback;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        sellerProfileDetails.setSellerNickname("TestNickname");
        sellerProfileDetails.setOverallFeedbackRating("4.5");
        sellerProfileDetails.setNumberOfFeedback("100");
        sellerProfileDetails.setNumberofCanceledAuctions("5");
        sellerProfileDetails.setStoreId("Store123");
        sellerProfileDetails.setStoreName("TestStore");
        sellerProfileDetails.setSellerFeedBack(sellerFeedback);
    }

    @Test
    void testToString() {
        when(sellerFeedback.toString()).thenReturn("MockFeedback");
        String expected = "NickName = TestNickname\n" + "OverallRating = 4.5\n" + "# of feedbacks = 100\n" + "# of Canceled Auctions = 5\n" + "StoreId = Store123\n" + "StoreName = TestStore\n" + "FeedBack = \nMockFeedback\n";
        String actual = sellerProfileDetails.toString();
        assertEquals(expected, actual);
    }
}
