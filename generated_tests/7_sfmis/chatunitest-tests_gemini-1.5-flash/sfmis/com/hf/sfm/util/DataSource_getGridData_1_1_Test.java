package com.hf.sfm.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import javax.servlet.http.HttpSession;

@ExtendWith(MockitoExtension.class)
class DataSource_getGridData_1_1_Test {

    @Mock
    private Loader loader;

    @InjectMocks
    private DataSource dataSource;

    @Test
    void testGetGridData_withPaging() {
        BasePara basePara = new BasePara();
        basePara.setSqlpath("testPath");
        basePara.setPaging(true);
        basePara.setStart(0);
        basePara.setLimit(10);
        List<Object> data = new ArrayList<>();
        data.add("test");
        ListRange range = new ListRange();
        range.setData(data);
        range.setTotalSize(100);
        when(loader.getRange()).thenReturn(range);
        ListRange result = dataSource.getGridData(basePara);
        assertEquals(data, result.getData());
        assertEquals(100, result.getTotalSize());
    }

    @Test
    void testGetGridData_withoutPaging() {
        BasePara basePara = new BasePara();
        basePara.setSqlpath("testPath");
        basePara.setPaging(false);
        basePara.setStart(0);
        basePara.setLimit(10);
        List<Object> data = new ArrayList<>();
        ListRange range = new ListRange();
        range.setData(data);
        range.setTotalSize(10);
        when(loader.getRange()).thenReturn(range);
        ListRange result = dataSource.getGridData(basePara);
        assertEquals(data, result.getData());
        assertEquals(10, result.getTotalSize());
    }

    @Test
    void testGetGridData_withEmptyData() {
        BasePara basePara = new BasePara();
        basePara.setSqlpath("testPath");
        basePara.setPaging(true);
        basePara.setStart(0);
        basePara.setLimit(10);
        List<Object> data = new ArrayList<>();
        ListRange range = new ListRange();
        range.setData(data);
        range.setTotalSize(0);
        when(loader.getRange()).thenReturn(range);
        ListRange result = dataSource.getGridData(basePara);
        assertEquals(data, result.getData());
        assertEquals(0, result.getTotalSize());
    }

    static class BasePara {

        private String sqlpath;

        private boolean paging;

        private int start;

        private int limit;

        // ... other fields ...
        public String getSqlpath() {
            return sqlpath;
        }

        public void setSqlpath(String sqlpath) {
            this.sqlpath = sqlpath;
        }

        public boolean isPaging() {
            return paging;
        }

        public void setPaging(boolean paging) {
            this.paging = paging;
        }

        public int getStart() {
            return start;
        }

        public void setStart(int start) {
            this.start = start;
        }

        public int getLimit() {
            return limit;
        }

        public void setLimit(int limit) {
            this.limit = limit;
        }
        // ... other getters and setters ...
    }

    static class ListRange {

        private List<Object> data;

        private int totalSize;

        public List<Object> getData() {
            return data;
        }

        public void setData(List<Object> data) {
            this.data = data;
        }

        public int getTotalSize() {
            return totalSize;
        }

        public void setTotalSize(int totalSize) {
            this.totalSize = totalSize;
        }
    }

    static class DataSource {

        private final Loader loader;

        DataSource(Loader loader) {
            this.loader = loader;
        }

        ListRange getGridData(BasePara basePara) {
            return loader.getRange();
        }
    }

    static class Loader {

        ListRange getRange() {
            return new ListRange();
        }
    }
}
