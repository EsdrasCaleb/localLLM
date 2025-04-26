package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class SellerProfileDetails_toString_14_4_Test {

    @Test
    void testToString() {
        SellerProfileDetails sellerProfileDetails = new SellerProfileDetails();
        String actual = sellerProfileDetails.toString();
        String expected = "NickName = \n" + "OverallRating = \n" + "# of feedbacks = \n" + "# of Canceled Auctions = \n" + "StoreId = \n" + "StoreName = \n" + "FeedBack = \n";
        assertEquals(expected, actual);
    }
}
