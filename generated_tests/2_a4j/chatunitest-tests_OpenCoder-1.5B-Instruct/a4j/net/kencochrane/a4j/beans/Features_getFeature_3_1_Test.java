// Test method
package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

public class Features_getFeature_3_1_Test {

    // Focal class
    public class Features {

        ArrayList<String> features;

        // Signatures of other methods defined in the focal class
        public String[] getFeature() {
            String[] retString = new String[features.size()];
            if (features.size() > 0)
                features.toArray(retString);
            return retString;
        }

        public void setFeature(String[] newString) {
            features = new ArrayList<>(Arrays.asList(newString));
        }

        public ArrayList<String> getFeaturesArray() {
            return features;
        }
    }

    // Test class
    public class FeaturesTest {

        @Test
        public void testGetFeature() {
            Features features = mock(Features.class);
            when(features.getFeature()).thenReturn(new String[] { "Feature1", "Feature2", "Feature3" });
            assertEquals("Feature1", features.getFeature()[0]);
            assertEquals("Feature2", features.getFeature()[1]);
            assertEquals("Feature3", features.getFeature()[2]);
            assertNull(features.getFeature()[-1]);
            assertNull(features.getFeature()[3]);
        }
    }
}
