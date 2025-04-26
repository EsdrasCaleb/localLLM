package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Features_toString_4_3_Test {

    @Test
    public void testToString() {
        Features features = new Features();
        features.setFeature(new String[] { "feature1", "feature2", "feature3" });
        features.setFeature(new String[] { "feature4", "feature5", "feature6" });
        String expectedOutput = "# of Feature = 3\nFeature - feature1\nFeature - feature2\nFeature - feature3\nFeature - feature4\nFeature - feature5\nFeature - feature6";
        assertEquals(expectedOutput, features.toString());
    }
}
