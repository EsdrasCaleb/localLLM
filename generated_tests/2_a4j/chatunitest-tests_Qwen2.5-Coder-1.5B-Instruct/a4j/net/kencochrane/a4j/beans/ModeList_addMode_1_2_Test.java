package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class ModeList_addMode_1_2_Test {

    ModeList modeList;

    @BeforeEach
    public void setUp() throws Exception {
        // Initialize the mode list instance
        modeList = new ModeList();
    }

    @Test
    public void testAddMode() {
        // Create an instance of Mode for testing
        Mode mode = mock(Mode.class);
        // Call the method to be tested
        modeList.addMode(mode);
        // Verify that the mode was added to the modes list
        assertEquals(1, modeList.getAllModes().size());
        // Assuming equals method exists in Mode class
        verify(mode).equals(mode);
    }
}
