package net.kencochrane.a4j.beans;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SellerProfile_toString_2_1_Test {

    @Test
    public void testToString() throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {
        // Positive case:  sellerProfileDetails has a valid toString()
        SellerProfileDetails sellerProfileDetailsMock = Mockito.mock(SellerProfileDetails.class);
        Mockito.when(sellerProfileDetailsMock.toString()).thenReturn("Mock Seller Details");
        SellerProfile sellerProfile = new SellerProfile();
        sellerProfile.setSellerProfileDetails(sellerProfileDetailsMock);
        String expectedOutput = "Mock Seller Details\n";
        String actualOutput = sellerProfile.toString();
        assertEquals(expectedOutput, actualOutput);
        // Negative case: sellerProfileDetails is null
        sellerProfile = new SellerProfile();
        sellerProfile.setSellerProfileDetails(null);
        // Empty String
        String expectedOutputNull = "";
        String actualOutputNull = sellerProfile.toString();
        assertEquals(expectedOutputNull, actualOutputNull);
    }
}

// SellerProfileDetails class
class SellerProfileDetails {

    @Override
    public String toString() {
        return "Seller Details";
    }
}
