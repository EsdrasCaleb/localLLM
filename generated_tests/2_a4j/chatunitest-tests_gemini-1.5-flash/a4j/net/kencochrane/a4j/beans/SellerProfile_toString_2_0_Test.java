package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
class SellerProfile_toString_2_0_Test {

    @Test
    void testToString_nullSellerProfileDetails() {
        SellerProfile sellerProfile = new SellerProfile();
        sellerProfile.setSellerProfileDetails(null);
        String expected = "null\n";
        assertEquals(expected, sellerProfile.toString());
    }

    @Test
    void testToString_validSellerProfileDetails() {
        SellerProfileDetails details = Mockito.mock(SellerProfileDetails.class);
        Mockito.when(details.toString()).thenReturn("mocked details");
        SellerProfile sellerProfile = new SellerProfile();
        sellerProfile.setSellerProfileDetails(details);
        String expected = "mocked details\n";
        assertEquals(expected, sellerProfile.toString());
    }

    @Test
    void testToString_emptySellerProfileDetails() {
        SellerProfileDetails details = new SellerProfileDetails();
        SellerProfile sellerProfile = new SellerProfile();
        sellerProfile.setSellerProfileDetails(details);
        String expected = "SellerProfileDetails{name='', address='', contactNumber=''}\n";
        assertEquals(expected, sellerProfile.toString());
    }

    @Test
    void testToString_populatedSellerProfileDetails() {
        SellerProfileDetails details = new SellerProfileDetails();
        details.setName("John Doe");
        details.setAddress("123 Main St");
        details.setContactNumber("555-1212");
        SellerProfile sellerProfile = new SellerProfile();
        sellerProfile.setSellerProfileDetails(details);
        String expected = "SellerProfileDetails{name='John Doe', address='123 Main St', contactNumber='555-1212'}\n";
        assertEquals(expected, sellerProfile.toString());
    }

    static class SellerProfileDetails {

        private String name = "";

        private String address = "";

        private String contactNumber = "";

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public String getAddress() {
            return address;
        }

        public void setAddress(String address) {
            this.address = address;
        }

        public String getContactNumber() {
            return contactNumber;
        }

        public void setContactNumber(String contactNumber) {
            this.contactNumber = contactNumber;
        }

        @Override
        public String toString() {
            return "SellerProfileDetails{name='" + name + '\'' + ", address='" + address + '\'' + ", contactNumber='" + contactNumber + '\'' + '}';
        }
    }

    static class SellerProfile {

        private SellerProfileDetails sellerProfileDetails;

        public void setSellerProfileDetails(SellerProfileDetails sellerProfileDetails) {
            this.sellerProfileDetails = sellerProfileDetails;
        }

        @Override
        public String toString() {
            return (sellerProfileDetails == null) ? "null\n" : sellerProfileDetails.toString() + "\n";
        }
    }
}
