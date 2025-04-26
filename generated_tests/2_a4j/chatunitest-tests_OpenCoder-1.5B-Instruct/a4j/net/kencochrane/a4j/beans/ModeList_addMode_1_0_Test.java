package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class ModeList_addMode_1_0_Test {

    @Test
    public void testAddMode() {
        // Create an instance of ModeList
        ModeList modeList = new ModeList();
        // Create a Mode object
        Mode mode = new Mode();
        // Add the mode using the addMode method
        modeList.addMode(mode);
        // Verify that the mode has been added to the list
        assertEquals(1, modeList.getAllModes().size());
    }
}
