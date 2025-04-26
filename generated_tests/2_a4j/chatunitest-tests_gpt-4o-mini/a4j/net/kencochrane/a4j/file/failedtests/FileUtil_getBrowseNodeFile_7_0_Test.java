package net.kencochrane.a4j.file;

import java.io.File;
import static org.mockito.ArgumentMatchers.anyString;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.data.Query;
import net.kencochrane.a4j.util.LoadProperties;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.Properties;
import java.util.Random;

class FileUtil_getBrowseNodeFile_7_0_Test {

    private FileUtil fileUtil;

    @BeforeEach
    void setUp() {
        fileUtil = Mockito.spy(new FileUtil());
        fileUtil.cacheDir = "/mock/cache/dir/";
        // example age
        fileUtil.oldestAge = 1000;
    }

    @Test
    void testGetBrowseNodeFile_CachedFileExistsAndIsValid() {
        String mode = "mode1";
        String node = "node1";
        String page = "page1";
        File cachedFile = mock(File.class);
        when(cachedFile.exists()).thenReturn(true);
        when(cachedFile.getPath()).thenReturn(fileUtil.cacheDir + mode + "_" + node + "_" + page + ".xml");
        when(fileUtil.isAgeGood(cachedFile)).thenReturn(true);
        doReturn(cachedFile).when(fileUtil).getBrowseNodeFile(mode, node, page);
        File result = fileUtil.getBrowseNodeFile(mode, node, page);
        assertEquals(cachedFile, result);
        verify(fileUtil, never()).downloadBrowseNodeFile(anyString(), anyString(), anyString(), anyString());
    }

    @Test
    void testGetBrowseNodeFile_CachedFileExistsAndIsOld() {
        String mode = "mode1";
        String node = "node1";
        String page = "page1";
        File cachedFile = mock(File.class);
        when(cachedFile.exists()).thenReturn(true);
        when(cachedFile.getPath()).thenReturn(fileUtil.cacheDir + mode + "_" + node + "_" + page + ".xml");
        when(fileUtil.isAgeGood(cachedFile)).thenReturn(false);
        doReturn(cachedFile).when(fileUtil).getBrowseNodeFile(mode, node, page);
        when(fileUtil.downloadBrowseNodeFile(mode, node, page, "t_" + mode + "_" + node + "_" + page + ".xml")).thenReturn(true);
        File result = fileUtil.getBrowseNodeFile(mode, node, page);
        assertEquals(cachedFile, result);
        verify(fileUtil).deleteFile(cachedFile.getPath());
        verify(fileUtil).renameFile("t_" + mode + "_" + node + "_" + page + ".xml", cachedFile.getPath());
    }

    @Test
    void testGetBrowseNodeFile_CachedFileDoesNotExistAndDownloadSucceeds() {
        String mode = "mode1";
        String node = "node1";
        String page = "page1";
        when(fileUtil.downloadBrowseNodeFile(mode, node, page, "t_" + mode + "_" + node + "_" + page + ".xml")).thenReturn(true);
        File result = fileUtil.getBrowseNodeFile(mode, node, page);
        assertEquals(new File(fileUtil.cacheDir + mode + "_" + node + "_" + page + ".xml"), result);
        verify(fileUtil).renameFile("t_" + mode + "_" + node + "_" + page + ".xml", fileUtil.cacheDir + mode + "_" + node + "_" + page + ".xml");
    }

    @Test
    void testGetBrowseNodeFile_CachedFileDoesNotExistAndDownloadFails() {
        String mode = "mode1";
        String node = "node1";
        String page = "page1";
        when(fileUtil.downloadBrowseNodeFile(mode, node, page, "t_" + mode + "_" + node + "_" + page + ".xml")).thenReturn(false);
        File result = fileUtil.getBrowseNodeFile(mode, node, page);
        assertNull(result);
    }
}
